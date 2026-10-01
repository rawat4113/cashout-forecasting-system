"""
Evaluate on the untouched, chronologically-last test block and compare to baselines.

Models compared
  Random                    - lower bound
  Historical frequency      - "this ATM cluster has always been busy" (static hotspot list)
  Recent activity (7d)      - "wherever complaints came from last week" (what an analyst does by hand)
  Logistic Regression       - simple learned model
  Gradient Boosting (ours)  - main model

Operational metrics (per day, averaged over test days; k = number of cells agencies can cover)
  precision@k        share of the k flagged cells where a cash-out really happened
  hotspot hit rate@k share of the day's k truly busiest cells that were flagged
  event coverage@k   share of all that day's cash-out events that fell inside the k flagged cells
  amount coverage@k  same, weighted by rupee value
  lead time          hours between the 00:00 alert and each covered withdrawal
                     vs. the reactive pipeline, which hears about a withdrawal only after the complaint arrives

Run:  python -m src.evaluate
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config as C
from .common import load_panel, split_panel
from .features import FEATURE_COLUMNS

MODEL_LABELS = {
    "random": "Random",
    "hist_freq": "Historical frequency",
    "recent7": "Recent activity (7d)",
    "logreg": "Logistic Regression",
    "hgb": "Gradient Boosting (ours)",
}
COLORS = {"random": "#9e9e9e", "hist_freq": "#8d6e63", "recent7": "#fb8c00",
          "logreg": "#1e88e5", "hgb": "#2e7d32"}


def _tie_break(score, rng):
    return np.asarray(score, float) + rng.random(len(score)) * 1e-9


def ranking_metrics(test: pd.DataFrame, score: np.ndarray, ks, rng) -> pd.DataFrame:
    df = test[["day_idx", "cell_id", "y", "y_cnt", "y_amt"]].copy()
    df["s"] = _tie_break(score, rng)
    rows = []
    for day, g in df.groupby("day_idx"):
        pred_order = g.sort_values("s", ascending=False)
        truth = g[g["y_cnt"] > 0].assign(t=lambda x: x["y_cnt"] + rng.random(len(x)) * 1e-6)
        truth = truth.sort_values("t", ascending=False)
        tot_ev, tot_amt = g["y_cnt"].sum(), g["y_amt"].sum()
        for k in ks:
            top = pred_order.head(k)
            true_top = set(truth.head(k)["cell_id"])
            rows.append(dict(
                day_idx=day, k=k,
                precision=top["y"].mean(),
                hotspot_hit_rate=len(true_top & set(top["cell_id"])) / max(len(true_top), 1) if true_top else np.nan,
                event_coverage=top["y_cnt"].sum() / tot_ev if tot_ev else np.nan,
                amount_coverage=top["y_amt"].sum() / tot_amt if tot_amt else np.nan))
    return pd.DataFrame(rows)


def lead_time_hours(test: pd.DataFrame, score: np.ndarray, events: pd.DataFrame, k: int, rng) -> np.ndarray:
    """Hours between the daily alert and every withdrawal that happens inside the flagged cells."""
    start = pd.Timestamp(C.START_DATE)
    df = test[["day_idx", "cell_id"]].copy()
    df["s"] = _tie_break(score, rng)
    flagged = (df.sort_values(["day_idx", "s"], ascending=[True, False])
                 .groupby("day_idx").head(k)[["day_idx", "cell_id"]])
    ev = events.copy()
    ev["day_idx"] = (ev["withdrawal_time"].dt.normalize() - start).dt.days
    hit = ev.merge(flagged, on=["day_idx", "cell_id"], how="inner")
    alert_time = start + pd.to_timedelta(hit["day_idx"], unit="D") + pd.Timedelta(hours=C.ALERT_HOUR)
    return ((hit["withdrawal_time"] - alert_time).dt.total_seconds() / 3600).to_numpy()


def build_scores(test: pd.DataFrame, rng) -> dict:
    hgb = joblib.load(C.MODEL_DIR / "hgb_model.joblib")
    logreg = joblib.load(C.MODEL_DIR / "logreg_model.joblib")
    return {
        "random": rng.random(len(test)),
        "hist_freq": test["hist_rate"].to_numpy(),
        "recent7": test["r7"].to_numpy(),
        "logreg": logreg.predict_proba(test[FEATURE_COLUMNS])[:, 1],
        "hgb": hgb.predict_proba(test[FEATURE_COLUMNS])[:, 1],
    }, hgb


def main():
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.SEED)

    events, cells, panel = load_panel()
    _, _, test = split_panel(panel)
    test_days = test["day_idx"].unique()
    scores, hgb = build_scores(test, rng)

    ks = list(C.TOP_K_LIST)
    ref_k = 10 if 10 in ks else ks[len(ks) // 2]
    ev_test = events[(events["withdrawal_time"].dt.normalize() - pd.Timestamp(C.START_DATE)).dt.days.isin(test_days)]
    reactive_lag = ((ev_test["complaint_time"] - ev_test["withdrawal_time"]).dt.total_seconds() / 3600)
    reactive_lag = reactive_lag[ev_test["traced"]]

    summary, daily_all = [], []
    for key, sc in scores.items():
        rm = ranking_metrics(test, sc, ks, rng)
        rm["model"] = key
        daily_all.append(rm)
        agg = rm.groupby("k")[["precision", "hotspot_hit_rate", "event_coverage", "amount_coverage"]].mean()
        lead = lead_time_hours(test, sc, ev_test, ref_k, rng)
        row = {"model": MODEL_LABELS[key],
               "PR-AUC": average_precision_score(test["y"], _tie_break(sc, rng)),
               "ROC-AUC": roc_auc_score(test["y"], _tie_break(sc, rng))}
        for k in ks:
            row[f"Precision@{k}"] = agg.loc[k, "precision"]
        for k in ks:
            row[f"HotspotHit@{k}"] = agg.loc[k, "hotspot_hit_rate"]
        for k in ks:
            row[f"EventCov@{k}"] = agg.loc[k, "event_coverage"]
        row[f"AmountCov@{ref_k}"] = agg.loc[ref_k, "amount_coverage"]
        row[f"MeanLeadTime_h@{ref_k}"] = float(np.mean(lead)) if len(lead) else np.nan
        if key in ("logreg", "hgb"):
            row["Brier"] = brier_score_loss(test["y"], sc)
        summary.append(row)

    summary = pd.DataFrame(summary).set_index("model")
    summary.round(4).to_csv(C.OUT_DIR / "metrics_summary.csv")
    pd.concat(daily_all).to_csv(C.OUT_DIR / "daily_metrics.csv", index=False)

    ours, recent = summary.loc[MODEL_LABELS["hgb"]], summary.loc[MODEL_LABELS["recent7"]]
    headline = {
        "test_days": int(len(test_days)),
        "test_positive_rate": float(test["y"].mean()),
        "reference_k": ref_k,
        "ours_precision_at_k": float(ours[f"Precision@{ref_k}"]),
        "recent7_precision_at_k": float(recent[f"Precision@{ref_k}"]),
        "random_precision_at_k": float(summary.loc[MODEL_LABELS["random"], f"Precision@{ref_k}"]),
        "ours_event_coverage_at_k": float(ours[f"EventCov@{ref_k}"]),
        "recent7_event_coverage_at_k": float(recent[f"EventCov@{ref_k}"]),
        "forecast_mean_lead_time_h": float(ours[f"MeanLeadTime_h@{ref_k}"]),
        "reactive_mean_lag_h": float(reactive_lag.mean()),
        "reactive_median_lag_h": float(reactive_lag.median()),
    }
    (C.OUT_DIR / "metrics.json").write_text(json.dumps(headline, indent=2))

    make_plots(test, scores, daily_all, hgb, events, cells, rng)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n=== TEST RESULTS (chronologically last %d days, never seen in training) ===" % len(test_days))
    print(summary.round(3).T.to_string())
    print("\n=== HEADLINE ===")
    for k, v in headline.items():
        print(f"{k:32s} {v:.3f}" if isinstance(v, float) else f"{k:32s} {v}")
    print(f"\nplots -> {C.PLOT_DIR}\nmetrics -> {C.OUT_DIR}")


def make_plots(test, scores, daily_all, hgb, events, cells, rng):
    # 1) precision@k and event coverage@k curves --------------------------------
    curve_ks = list(range(1, 31))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, sc in scores.items():
        rm = ranking_metrics(test, sc, curve_ks, rng).groupby("k")[["precision", "event_coverage"]].mean()
        axes[0].plot(rm.index, rm["precision"], label=MODEL_LABELS[key], color=COLORS[key], lw=2.2 if key == "hgb" else 1.5)
        axes[1].plot(rm.index, rm["event_coverage"], label=MODEL_LABELS[key], color=COLORS[key], lw=2.2 if key == "hgb" else 1.5)
    axes[0].set(title="Precision@k (flagged cells with a real cash-out)", xlabel="k = cells flagged per day", ylabel="precision")
    axes[1].set(title="Event coverage@k (share of cash-outs inside flagged cells)", xlabel="k = cells flagged per day", ylabel="coverage")
    for a in axes:
        a.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.PLOT_DIR / "precision_coverage_at_k.png", dpi=150)
    plt.close(fig)

    # 2) feature importance (permutation, PR-AUC drop) ----------------------------
    sample = test.sample(min(8000, len(test)), random_state=C.SEED)
    imp = permutation_importance(hgb, sample[FEATURE_COLUMNS], sample["y"], scoring="average_precision",
                                 n_repeats=5, random_state=C.SEED)
    order = np.argsort(imp.importances_mean)[::-1][:15][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(FEATURE_COLUMNS)[order], imp.importances_mean[order], xerr=imp.importances_std[order], color="#2e7d32")
    ax.set(title="Top features (permutation importance, drop in PR-AUC)", xlabel="importance")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(C.PLOT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": imp.importances_mean}) \
        .sort_values("importance", ascending=False).to_csv(C.OUT_DIR / "feature_importance.csv", index=False)

    # 3) lead-time comparison ----------------------------------------------------
    ref_k = 10
    start = pd.Timestamp(C.START_DATE)
    days = test["day_idx"].unique()
    ev_test = events[(events["withdrawal_time"].dt.normalize() - start).dt.days.isin(days)]
    lead = lead_time_hours(test, scores["hgb"], ev_test, ref_k, rng)
    lag_h = ((ev_test["complaint_time"] - ev_test["withdrawal_time"]).dt.total_seconds() / 3600)[ev_test["traced"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(-lag_h.clip(upper=72), bins=48, alpha=.65, color="#c62828", label=f"Reactive: complaint arrives AFTER withdrawal (mean {lag_h.mean():.0f} h late; all traced cash-outs)")
    ax.hist(lead, bins=24, alpha=.65, color="#2e7d32", label=f"Forecast alert issued BEFORE withdrawal (mean {lead.mean():.0f} h early; covers {len(lead)/len(ev_test):.0%} of all cash-outs)")
    ax.axvline(0, color="k", lw=1)
    ax.set(title=f"Advance warning vs. reactive pipeline (top-{ref_k} cells/day)", xlabel="hours relative to the withdrawal (negative = after the money is gone)", ylabel="events")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.PLOT_DIR / "lead_time.png", dpi=150)
    plt.close(fig)

    # 4) risk map for one test day -------------------------------------------------
    day = int(test.groupby("day_idx")["y_cnt"].sum().idxmax())
    d = test[test["day_idx"] == day].merge(cells[["cell_id", "city"]], on="cell_id")
    d["risk"] = scores["hgb"][test["day_idx"].to_numpy() == day]
    top = d.nlargest(10, "risk")
    fig, ax = plt.subplots(figsize=(6.5, 7))
    sc = ax.scatter(d["lon"], d["lat"], c=d["risk"], s=25 + 25 * d["risk"] * 8, cmap="YlOrRd", edgecolor="k", linewidth=.3, vmin=0, vmax=1)
    ax.scatter(top["lon"], top["lat"], facecolors="none", edgecolors="#1565c0", s=190, linewidth=2, label="Top-10 flagged")
    real = d[d["y"] == 1]
    ax.scatter(real["lon"], real["lat"], marker="x", c="k", s=30, linewidth=1, label="Actual cash-out")
    fig.colorbar(sc, label="predicted cash-out probability", shrink=.7)
    ax.set(title=f"Risk map, {pd.Timestamp(d['date'].iloc[0]).date()} (busiest test day)", xlabel="longitude", ylabel="latitude")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(C.PLOT_DIR / "risk_map_example.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
