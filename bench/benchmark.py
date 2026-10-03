"""
Honest micro-benchmark for the real-time path. Run it on ANY machine, then run it again on the IBM Z /
LinuxONE LPAR you get for the Datathon, and put the two JSON files side by side:

    python bench/benchmark.py                  # writes outputs/benchmarks/bench_<arch>_<host>.json

What is measured (single process, synthetic data, no network):
  ingest      : validate + update online state for one complaint event            (events/second)
  score       : online features + model inference for ALL clusters, one forecast  (milliseconds)
  scaling     : the same forecast with 145 / 1,000 / 3,000 synthetic clusters
Nothing here is an IBM Z claim until you run it on IBM Z.
"""
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                      # noqa: E402
from src.common import load_data                 # noqa: E402
from src.stream.pipeline import StreamPipeline   # noqa: E402


def pct(a, q):
    return round(float(np.percentile(a, q)), 3)


def synth_cells(n, seed=0):
    rng = np.random.default_rng(seed)
    n_city = max(n // 8, 1)
    centre = np.column_stack([rng.uniform(8, 32, n_city), rng.uniform(70, 88, n_city)])
    city = rng.integers(0, n_city, n)
    pts = centre[city] + rng.normal(0, 0.15, (n, 2))
    return pd.DataFrame(dict(cell_id=np.arange(n), city=[f"city{c}" for c in city], state="S", lat=pts[:, 0], lon=pts[:, 1],
                             tier=rng.integers(1, 4, n), atm_count=rng.integers(20, 90, n)))


def bench_scoring(cells, events_per_day, n_runs, tmp):
    p = StreamPipeline(cells, audit_path=tmp / f"a{len(cells)}.jsonl")
    rng = np.random.default_rng(1)
    d0 = pd.Timestamp("2025-06-01")
    n_hist = int(events_per_day * 500)
    hist = pd.DataFrame(dict(
        event_id=np.arange(n_hist), cell_id=rng.integers(0, len(cells), n_hist),
        complaint_time=pd.Timestamp(C.START_DATE) + pd.to_timedelta(rng.uniform(0, 500 * 86400, n_hist), unit="s"),
        category=rng.choice(["UPI / Card Fraud", "Digital Arrest Scam", "KYC / OTP Fraud"], n_hist),
        amount=rng.uniform(1e3, 3e5, n_hist), traced=True))
    p.store.bulk_load(hist, before=d0)
    feat, inf, tot = [], [], []
    for _ in range(n_runs):
        t = {}
        p.scorer.score_day(d0, timings=t)
        feat.append(t["feature_ms"]); inf.append(t["inference_ms"]); tot.append(t["total_ms"])
    return dict(clusters=len(cells), history_events=n_hist, feature_ms_p50=pct(feat, 50), inference_ms_p50=pct(inf, 50),
                total_ms_p50=pct(tot, 50), total_ms_p95=pct(tot, 95), total_ms_p99=pct(tot, 99),
                clusters_scored_per_s=round(len(cells) / (np.median(tot) / 1000)))


def main():
    tmp = C.OUT_DIR / "benchmarks"
    tmp.mkdir(parents=True, exist_ok=True)
    events, cells = load_data()

    # --- ingest throughput on real replayed events ------------------------------------------------
    p = StreamPipeline(cells, audit_path=tmp / "ingest.jsonl")
    live = events[events["traced"]].sort_values("complaint_time")
    recs = live[["event_id", "complaint_time", "cell_id", "category", "amount"]].to_dict("records") * 3
    t0 = time.perf_counter()
    for r in recs:
        p.ingest(r)
    dt = time.perf_counter() - t0
    ingest = dict(events=len(recs), seconds=round(dt, 3), events_per_s=round(len(recs) / dt), **{k: v for k, v in p.ingest_latency.summary().items() if k != "count"})

    scoring = [bench_scoring(cells, 40, 60, tmp)] + [bench_scoring(synth_cells(n), 40 * n / 145, 15, tmp) for n in (1000, 3000)]
    out = dict(
        platform=dict(machine=platform.machine(), byteorder=sys.byteorder, system=platform.platform(), python=platform.python_version(),
                      numpy=np.__version__, cpus=os.cpu_count(), host=socket.gethostname()),
        note="synthetic data, single process, no network; numbers are only meaningful for the machine named in 'platform'",
        ingest=ingest, scoring=scoring)
    path = tmp / f"bench_{platform.machine()}_{socket.gethostname()}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nwritten:", path)


if __name__ == "__main__":
    main()
