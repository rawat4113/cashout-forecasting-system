"""Shared helpers: data loading and the chronological train/val/test split."""
import pandas as pd

from . import config as C
from .features import build_panel


def load_data():
    events = pd.read_csv(C.DATA_DIR / "events.csv", parse_dates=["withdrawal_time", "complaint_time"])
    cells = pd.read_csv(C.DATA_DIR / "cells.csv")
    return events, cells


def split_bounds(n_days: int = C.N_DAYS):
    train_end = int(n_days * C.TRAIN_FRAC)
    val_end = int(n_days * (C.TRAIN_FRAC + C.VAL_FRAC))
    return train_end, val_end


def split_panel(panel: pd.DataFrame, n_days: int = C.N_DAYS):
    """Chronological split. Never shuffle time-series data."""
    train_end, val_end = split_bounds(n_days)
    tr = panel[panel["day_idx"] < train_end]
    va = panel[(panel["day_idx"] >= train_end) & (panel["day_idx"] < val_end)]
    te = panel[panel["day_idx"] >= val_end]
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def load_panel():
    events, cells = load_data()
    return events, cells, build_panel(events, cells)
