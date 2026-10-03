"""Alert engine: ranked scores -> de-duplicated, explained, human-review alerts."""
from __future__ import annotations

import pandas as pd

from .. import config as C

ACTION = {
    "HIGH": "Priority human review: share cluster with partner-bank fraud desks and LEA/I4C duty officer for a verification decision.",
    "MEDIUM": "Watchlist: re-check at next refresh; analyst may escalate.",
    "LOW": "No action.",
}
_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def reason_codes(row: pd.Series, day: pd.DataFrame) -> list[str]:
    """Plain-language drivers computed from the feature values (heuristic explanation, not SHAP)."""
    why = []
    if row["burst"] >= 1.5 and row["r3"] >= 2:
        why.append(f"Complaint activity accelerating (3-day rate {row['burst']:.1f}x the 28-day baseline)")
    if row["r7"] >= day["r7"].quantile(0.90) and row["r7"] > 0:
        why.append(f"{int(row['r7'])} traced complaints in 7 days (top 10% of all clusters today)")
    if row["nbr7"] >= day["nbr7"].quantile(0.90) and row["nbr7"] > 0:
        why.append(f"Neighbouring clusters also active ({int(row['nbr7'])} complaints in 7 days)")
    if row["hv7"] > 0:
        why.append(f"{int(row['hv7'])} high-value scam complaint(s) in 7 days")
    if row["days_since_last"] <= 1:
        why.append("A complaint was traced here within the last day")
    return why or ["Elevated score from combined history (no single dominant driver)"]


class AlertEngine:
    """Keeps per-(cluster, day) state so repeated refreshes produce NEW / ESCALATED / UNCHANGED, not spam."""

    def __init__(self):
        self._seen: dict[tuple[int, str], str] = {}

    def build(self, table: pd.DataFrame, date, min_level: str = "MEDIUM", model_sha: str = "") -> list[dict]:
        stamp = pd.Timestamp(date).strftime("%Y-%m-%d")
        out = []
        for _, r in table.iterrows():
            if _ORDER[r["risk_level"]] < _ORDER[min_level]:
                continue
            key = (int(r["cell_id"]), stamp)
            prev = self._seen.get(key)
            if prev is None:
                status = "NEW"
            elif _ORDER[r["risk_level"]] > _ORDER[prev]:
                status = "ESCALATED"
            else:
                status = "UNCHANGED"
            self._seen[key] = r["risk_level"]
            out.append(dict(
                alert_id=f"{stamp}-C{int(r['cell_id']):03d}", status=status, rank=int(r["rank"]),
                cell_id=int(r["cell_id"]), city=str(r["city"]), state=str(r["state"]),
                lat=float(r["lat"]), lon=float(r["lon"]),
                risk_score=round(float(r["risk_score"]), 4), level=str(r["risk_level"]),
                fraud_type=str(r["fraud_type"]), recent_complaints_7d=int(r["r7"]),
                reasons=reason_codes(r, table), recommended_action=ACTION[r["risk_level"]],
                human_review_required=True, model_sha256=model_sha,
            ))
        return out
