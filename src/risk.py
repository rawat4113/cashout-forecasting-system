"""Risk-level thresholds (kept dependency-free so the serving node does not need scikit-learn)."""
from . import config as C


def risk_level(p: float) -> str:
    if p >= C.RISK_HIGH:
        return "HIGH"
    if p >= C.RISK_MEDIUM:
        return "MEDIUM"
    return "LOW"
