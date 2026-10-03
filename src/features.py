"""
Feature engineering: turn the event log into a (cell x day) panel.

LEAKAGE RULE (the most important design decision in this project)
-----------------------------------------------------------------
Label for (cell c, day d)  = did a cash-out *happen* in c during day d?   (uses withdrawal_time)
Features for (cell c, day d) = only complaints whose complaint_time is BEFORE 00:00 of day d.
                               (uses complaint_time - what the portal actually knew)

Because complaints arrive ~20 h after the withdrawal and ~15% are never traced,
yesterday's cash-outs are only partly visible. The model must learn despite that,
exactly like a real deployment would.
"""
import numpy as np
import pandas as pd

from . import config as C

HIGH_VALUE_CATEGORIES = {"Digital Arrest Scam", "Investment / Trading Scam"}

FEATURE_COLUMNS = [
    # recent complaint activity in the cell
    "lag1", "lag2", "lag3", "lag7", "r3", "r7", "r14", "r28",
    "ewm", "burst", "days_since_last", "active_days_14", "hist_rate",
    "amt7_log", "hv7",
    # spill-over / context
    "nbr7", "city7_others", "nat_r7_share",
    # calendar
    "dow", "dom", "month_end", "t",
    # static cell attributes
    "lat", "lon", "atm_count", "tier",
]


def haversine_matrix(lat, lon) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def nearest_neighbours(cells: pd.DataFrame, k: int) -> np.ndarray:
    dist = haversine_matrix(cells["lat"].to_numpy(), cells["lon"].to_numpy())
    np.fill_diagonal(dist, np.inf)
    return np.argsort(dist, axis=1)[:, :k]


def window_sum(a: np.ndarray, w: int) -> np.ndarray:
    """out[:, d] = sum of a[:, max(0, d-w) : d]  (window ENDS the day before d)."""
    cs = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, axis=1)], axis=1)
    d = np.arange(a.shape[1])
    lo = np.maximum(0, d - w)
    return cs[:, d] - cs[:, lo]


def lag(a: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(a, dtype=float)
    out[:, k:] = a[:, :-k]
    return out


def _day_index(ts: pd.Series, start: pd.Timestamp) -> np.ndarray:
    return (ts.dt.normalize() - start).dt.days.to_numpy()


def build_arrays(events: pd.DataFrame, n_cells: int, start: pd.Timestamp, n_days: int):
    cid = events["cell_id"].to_numpy()
    w_day = _day_index(events["withdrawal_time"], start)
    c_day = _day_index(events["complaint_time"], start)
    amount = events["amount"].to_numpy(float)
    traced = events["traced"].to_numpy(bool)
    high_value = events["category"].isin(HIGH_VALUE_CATEGORIES).to_numpy()

    y_cnt = np.zeros((n_cells, n_days))
    y_amt = np.zeros((n_cells, n_days))
    m = (w_day >= 0) & (w_day < n_days)
    np.add.at(y_cnt, (cid[m], w_day[m]), 1.0)
    np.add.at(y_amt, (cid[m], w_day[m]), amount[m])

    o_cnt = np.zeros((n_cells, n_days))
    o_amt = np.zeros((n_cells, n_days))
    o_hv = np.zeros((n_cells, n_days))
    m = traced & (c_day >= 0) & (c_day < n_days)           # only complaints that reached the portal
    np.add.at(o_cnt, (cid[m], c_day[m]), 1.0)
    np.add.at(o_amt, (cid[m], c_day[m]), amount[m])
    np.add.at(o_hv, (cid[m & high_value], c_day[m & high_value]), 1.0)
    return y_cnt, y_amt, o_cnt, o_amt, o_hv


def features_from_arrays(o_cnt, o_amt, o_hv, cells: pd.DataFrame, start: pd.Timestamp, n_days: int,
                         nbr: np.ndarray = None) -> dict:
    """Feature dict (name -> (n_cells, n_days) array) from per-cell/day complaint arrays.

    This is the SINGLE implementation of the feature logic. The batch trainer
    (build_panel) and the real-time scorer (src/stream/online_store.py) both call it,
    so training-time and serving-time features cannot drift apart.
    """
    n_cells = len(cells)
    f = {}
    for k in (1, 2, 3, 7):
        f[f"lag{k}"] = lag(o_cnt, k)
    for w in (3, 7, 14, 28):
        f[f"r{w}"] = window_sum(o_cnt, w)
    f["amt7_log"] = np.log1p(window_sum(o_amt, 7))
    f["hv7"] = window_sum(o_hv, 7)
    f["active_days_14"] = window_sum((o_cnt > 0).astype(float), 14)

    ewm = np.zeros_like(o_cnt)
    dsl = np.zeros_like(o_cnt)
    last_seen = np.full(n_cells, -1000.0)
    for d in range(1, n_days):
        ewm[:, d] = 0.3 * o_cnt[:, d - 1] + 0.7 * ewm[:, d - 1]
        last_seen = np.where(o_cnt[:, d - 1] > 0, d - 1, last_seen)
        dsl[:, d] = np.minimum(d - last_seen, 60)
    dsl[:, 0] = 60
    f["ewm"], f["days_since_last"] = ewm, dsl
    f["burst"] = (f["r3"] + 0.5) / (f["r28"] * 3 / 28 + 0.5)       # is activity accelerating?

    days = np.arange(n_days)
    f["hist_rate"] = window_sum((o_cnt > 0).astype(float), n_days) / np.maximum(days, 1)[None, :]

    # spill-over: nearest geographic neighbours, and the rest of the same city
    r7 = f["r7"]
    if nbr is None:
        nbr = nearest_neighbours(cells, C.KNN_NEIGHBOURS)
    f["nbr7"] = r7[nbr].sum(axis=1)
    city_codes = cells["city"].astype("category").cat.codes.to_numpy()
    city_tot = np.zeros((city_codes.max() + 1, n_days))
    np.add.at(city_tot, city_codes, r7)
    f["city7_others"] = city_tot[city_codes] - r7
    f["nat_r7_share"] = r7 / np.maximum(r7.sum(axis=0, keepdims=True), 1)

    dates = pd.date_range(start, periods=n_days)
    cal = {
        "dow": dates.dayofweek.to_numpy(),
        "dom": dates.day.to_numpy(),
        "month_end": ((dates.day >= 28) | (dates.day <= 3)).astype(int),
        "t": days,
    }
    for name, arr in cal.items():
        f[name] = np.tile(arr, (n_cells, 1)).astype(float)
    for name in ("lat", "lon", "atm_count", "tier"):
        f[name] = np.repeat(cells[name].to_numpy(float)[:, None], n_days, axis=1)
    return f


def build_panel(events: pd.DataFrame, cells: pd.DataFrame,
                start: str = C.START_DATE, n_days: int = C.N_DAYS,
                warmup: int = C.WARMUP_DAYS) -> pd.DataFrame:
    """Return one row per (cell, day) with features + labels."""
    start = pd.Timestamp(start)
    n_cells = len(cells)
    y_cnt, y_amt, o_cnt, o_amt, o_hv = build_arrays(events, n_cells, start, n_days)
    f = features_from_arrays(o_cnt, o_amt, o_hv, cells, start, n_days)
    days = np.arange(n_days)
    dates = pd.date_range(start, periods=n_days)

    panel = pd.DataFrame({k: v.ravel() for k, v in f.items()})
    panel.insert(0, "cell_id", np.repeat(cells["cell_id"].to_numpy(), n_days))
    panel.insert(1, "day_idx", np.tile(days, n_cells))
    panel.insert(2, "date", np.tile(dates, n_cells))
    panel["y_cnt"] = y_cnt.ravel()
    panel["y_amt"] = y_amt.ravel()
    panel["y"] = (panel["y_cnt"] > 0).astype(int)
    panel = panel[panel["day_idx"] >= warmup].reset_index(drop=True)
    return panel[["cell_id", "day_idx", "date"] + FEATURE_COLUMNS + ["y", "y_cnt", "y_amt"]]
