"""
Train the cash-out hotspot models.

* HistGradientBoostingClassifier  -> main model (small hyper-parameter search on the validation block)
* Logistic Regression             -> simple learned reference model

Both predict P(cash-out happens in cell c on day d) from information available at 00:00 of day d.

Run:  python -m src.train
"""
import json

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .common import load_panel, split_panel
from .features import FEATURE_COLUMNS

PARAM_GRID = [
    dict(max_iter=100, max_depth=3),
    dict(max_iter=250, max_depth=3),
    dict(max_iter=100, max_depth=5),
    dict(max_iter=250, max_depth=5),
]


def make_hgb(**params):
    return HistGradientBoostingClassifier(
        learning_rate=0.05, min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=False,          # internal early stopping uses a RANDOM split -> not time-aware
        random_state=C.SEED, **params)


def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def main():
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _, _, panel = load_panel()
    train, val, test = split_panel(panel)
    print(f"rows  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    # 1) pick hyper-parameters on the validation block (chronologically after train)
    best, best_ap = None, -1.0
    for params in PARAM_GRID:
        model = make_hgb(**params).fit(train[FEATURE_COLUMNS], train["y"])
        ap = average_precision_score(val["y"], model.predict_proba(val[FEATURE_COLUMNS])[:, 1])
        print(f"  {params}  val PR-AUC = {ap:.4f}")
        if ap > best_ap:
            best, best_ap = params, ap
    print(f"best params: {best}  (val PR-AUC {best_ap:.4f})")

    # 2) refit on train+val with the chosen parameters; the test block stays untouched
    full = pd.concat([train, val], ignore_index=True)
    hgb = make_hgb(**best).fit(full[FEATURE_COLUMNS], full["y"])
    logreg = make_logreg().fit(full[FEATURE_COLUMNS], full["y"])

    joblib.dump(hgb, C.MODEL_DIR / "hgb_model.joblib")
    joblib.dump(logreg, C.MODEL_DIR / "logreg_model.joblib")
    meta = dict(features=FEATURE_COLUMNS, best_params=best, val_pr_auc=best_ap,
                n_train_rows=int(len(full)), seed=C.SEED)
    (C.MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"saved models -> {C.MODEL_DIR}")


if __name__ == "__main__":
    main()
