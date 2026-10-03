"""Batch/stream parity: the real-time path must reproduce the batch forecast exactly."""
import numpy as np
import pandas as pd

from src import config as C
from src.common import load_data
from src.features import FEATURE_COLUMNS, build_panel
from src.predict import forecast_from_frames
from src.stream.pipeline import StreamPipeline
from src.stream.schema import ComplaintEvent

DATES = ["2025-02-10", "2025-05-20", "2025-06-24"]


def _pipe(tmp_path, events, cells, date):
    p = StreamPipeline(cells, audit_path=tmp_path / "a.jsonl")
    p.store.bulk_load(events, before=pd.Timestamp(date))
    return p


def test_online_features_equal_batch_features(tmp_path):
    events, cells = load_data()
    panel = build_panel(events, cells, n_days=C.N_DAYS + 1)
    for d in DATES:
        day_idx = (pd.Timestamp(d) - pd.Timestamp(C.START_DATE)).days
        ref = panel[panel["day_idx"] == day_idx].sort_values("cell_id")[FEATURE_COLUMNS].to_numpy()
        f = _pipe(tmp_path, events, cells, d).store.features_for_day(d)
        got = np.column_stack([f[c] for c in FEATURE_COLUMNS])
        assert np.allclose(ref, got, atol=1e-9), f"features differ on {d}"


def test_event_by_event_ingest_equals_bulk(tmp_path):
    """Feeding complaints one by one (the live path) gives the same state as the bulk warm start."""
    events, cells = load_data()
    d = pd.Timestamp("2025-03-15")
    live = StreamPipeline(cells, audit_path=tmp_path / "b.jsonl")
    sub = events[events["traced"] & (events["complaint_time"] < d)].sort_values("complaint_time")
    for r in sub[["event_id", "complaint_time", "cell_id", "category", "amount"]].to_dict("records"):
        assert live.ingest(r)["accepted"]
    bulk = _pipe(tmp_path, events, cells, d)
    for k in ("o_cnt", "o_amt", "o_hv", "o_cat"):
        assert np.allclose(getattr(live.store, k), getattr(bulk.store, k))


def test_realtime_scores_equal_batch_forecast(tmp_path):
    events, cells = load_data()
    for d in DATES:
        ref = forecast_from_frames(events, cells, pd.Timestamp(d), top_k=None)
        got = _pipe(tmp_path, events, cells, d).scorer.score_day(d)[list(ref.columns)]
        assert (ref["cell_id"].to_numpy() == got["cell_id"].to_numpy()).all()
        assert np.allclose(ref["risk_score"], got["risk_score"], atol=1e-9)
        assert (ref["fraud_type"].to_numpy() == got["fraud_type"].to_numpy()).all()
        assert (ref["fraud_volume_7d"].to_numpy() == got["fraud_volume_7d"].to_numpy()).all()


def test_future_event_cannot_change_todays_forecast(tmp_path):
    events, cells = load_data()
    d = pd.Timestamp("2025-04-10")
    p = _pipe(tmp_path, events, cells, d)
    before = p.scorer.score_day(d)["risk_score"].to_numpy().copy()
    p.ingest(dict(event_id=-1, complaint_time=str(d + pd.Timedelta(hours=3)), cell_id=int(cells.cell_id[0]),
                  category="Digital Arrest Scam", amount=900000))
    after = p.scorer.score_day(d)["risk_score"].to_numpy()
    assert np.array_equal(before, after)


def test_bad_events_are_rejected(tmp_path):
    events, cells = load_data()
    p = StreamPipeline(cells, audit_path=tmp_path / "c.jsonl")
    assert not p.ingest({"event_id": 1})["accepted"]
    assert not p.ingest(dict(event_id=1, complaint_time="2025-01-01", cell_id=99999, category="x", amount=5))["accepted"]
    assert not p.ingest(dict(event_id=1, complaint_time="2025-01-01", cell_id=0, category="x", amount=-5))["accepted"]
