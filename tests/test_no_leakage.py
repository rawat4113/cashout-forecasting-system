"""
The single most important correctness test.

Features for day d must be identical whether or not we delete every complaint that arrives
at/after 00:00 of day d - i.e. the model never sees the future.
"""
import numpy as np
import pandas as pd

from src import config as C
from src.features import FEATURE_COLUMNS, build_panel
from src.generate_data import generate


def test_features_use_only_past_complaints():
    events, cells = generate(seed=1)
    start = pd.Timestamp(C.START_DATE)
    full = build_panel(events, cells)

    for day_idx in (60, 200, 400):
        cutoff = start + pd.Timedelta(days=day_idx)
        past_only = events[events["complaint_time"] < cutoff]
        trimmed = build_panel(past_only, cells)
        a = full[full["day_idx"] == day_idx][FEATURE_COLUMNS].to_numpy()
        b = trimmed[trimmed["day_idx"] == day_idx][FEATURE_COLUMNS].to_numpy()
        assert np.allclose(a, b), f"future information leaked into features of day {day_idx}"


def test_panel_shape_and_labels():
    events, cells = generate(seed=2)
    panel = build_panel(events, cells)
    assert len(panel) == len(cells) * (C.N_DAYS - C.WARMUP_DAYS)
    assert not panel[FEATURE_COLUMNS].isna().any().any()
    assert set(panel["y"].unique()) <= {0, 1}
    assert int(panel["y_cnt"].sum()) <= len(events)
